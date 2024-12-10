# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import threading
import pytest
from self_speculation.speculative_streamer import SpeculativeTextStreamer
import torch

@pytest.fixture
def streamer(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.decode = mocker.MagicMock(side_effect=lambda tokens, **kwargs: ' '.join(str(x) for x in tokens))
    return SpeculativeTextStreamer(tokenizer=tokenizer, non_blocking=False)

def test_put_tokens_non_blocking(streamer, mocker):
    mocker.patch('threading.Thread')
    streamer.non_blocking = True
    streamer.put([1, 2, 3])
    threading.Thread.assert_called_once()

def test_thread_safety(streamer, mocker):
    mocker.patch('threading.Thread')
    streamer.non_blocking = True
    for _ in range(10):
        streamer.put([1, 2, 3], is_draft=False)
    assert threading.Thread.call_count == 10

def test_put_tokens_blocking(streamer):
    # Convert list to tensor if necessary
    input_tensor = torch.tensor([1, 2, 3])
    streamer.put(input_tensor, is_draft=False)
    streamer.tokenizer.decode.assert_called_once_with([1, 2, 3], **streamer.decode_kwargs)
    assert streamer.text_cache == '1 2 3'

def test_delete_tokens(streamer):
    input_tensor = torch.tensor([1, 2, 3, 4, 5])
    streamer.put(input_tensor, is_draft=False)
    
    streamer.delete(2)
    assert streamer.text_cache[:streamer.print_len]

@pytest.mark.parametrize("is_draft", [True, False])
def test_draft_handling(streamer, is_draft):
    input_tensor = torch.tensor([1, 2, 3])
    streamer.put(input_tensor, is_draft=is_draft)
    if is_draft:
        assert '\n' not in streamer.text_cache
    else:
        assert streamer.text_cache == '1 2 3'
