from typing import List, Optional, Tuple

import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

class SelfSpeculativeGenerationStrategy(GenerationStrategy):
    # TODO implement stopping based on EOS token prediction
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
    ) -> GenerationStrategyResult:
        past_key_values = None

        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to("cuda")
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_id=eos_token_id,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            if eos_token_id in output_ids:
                # break out of loop when we get an EOS token
                # remove the EOS token id
                output_ids = output_ids[: output_ids.index(eos_token_id)]
                break
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations,
        )

    # TODO: remove calls, input_ids_list, rely on generation config
    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_id: int,
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
    ):
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        for _ in range(num_speculations):
            draft_result = forward_early(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            draft_next_token, draft_next_prob = decode_next_token(logits=draft_result.logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token == eos_token_id:
                # break out of loop when we get an EOS token
                break

        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make
        # 1 x (T_d  + T_p)
        prefill_token_ids = torch.cat(
            [input_ids, torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)],
            dim=-1,
        )

        # logits: 1 x (T_d  + T_p) x V
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
        )
        logits = verify_results.logits
        past_key_values = verify_results.past_key_values
        # only select the logits relevant to what the draft has outputted.
        # verification_logits: 1 x T_d x V
        verification_logits = logits[:, prompt_length - 1 :, :]

        # verified_tokens: 1 x (T_d)
        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        verified_tokens, _ = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)

        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = prefill_token_ids[:, prompt_length:] == verified_tokens[:, :-1]

        # number of matches is the index of the number of tokens we are accepting from the draft
        number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()

        # accept the `number_of_matches` tokens from the draft with one more from the main model
        # since we re-use the same cachem the input id should only be the last accepted token TODO check this
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        # input_ids = verified_tokens[:, : number_of_matches + 1]
        output_ids.extend(verified_tokens[0][: number_of_matches + 1].tolist())
        
        if False:
            r = np.random.uniform(size=len(draft_output_ids))
            filtered_verification_logits = transformers.top_k_top_p_filtering(verification_logits.flatten(0,1) / temperature, top_k=top_k, top_p=top_p).unsqueeze(0)
            verification_probabilities = torch.nn.functional.softmax(filtered_verification_logits, dim=-1)

            number_of_matches = 0
            for i in range(len(draft_output_ids)):
                if r[i] < min(1, verification_probabilities[0, i, draft_output_ids[i]].item() / draft_probabilities[i][0, draft_output_ids[i]].item()):
                    output_ids.append(draft_output_ids[i])
                    number_of_matches += 1
                else:
                    output_ids.append(torch.multinomial(max_fn((verification_probabilities[0, i, :] - draft_probabilities[i])), num_samples=1).item())
                    break
            else:
                i += 1
                verified_out = torch.multinomial(verification_probabilities[0, i, :], num_samples=1)
                output_ids.append(verified_out.item())
                # accept the `number_of_matches` tokens from the draft with one more from the main model
                # since we re-use the same cachem the input id should only be the last accepted token TODO check this
                input_ids = verified_out.unsqueeze(0)


        # we want the entire output sequence + input sequence
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            len(draft_output_ids),
        )
