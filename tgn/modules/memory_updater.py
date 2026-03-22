from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class TransformerMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device, num_heads=2, num_layers=1, dropout=0.1):
    super(TransformerMemoryUpdater, self).__init__()
    self.memory = memory
    self.message_dimension = message_dimension
    self.memory_dimension = memory_dimension
    self.device = device
    self.seq_len = memory.seq_len

    assert self.seq_len > 0, "TransformerMemoryUpdater requires memory to be initialized with seq_len > 0"

    self.message_proj = nn.Linear(message_dimension, memory_dimension) if message_dimension != memory_dimension else nn.Identity()

    encoder_layer = nn.TransformerEncoderLayer(d_model=memory_dimension, nhead=num_heads, dropout=dropout, batch_first=True)
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

  def _append_to_buffer(self, buffer, lengths, node_ids, messages):
    new_buffer = buffer.clone()
    new_lengths = lengths.clone()
    
    b = new_buffer[node_ids]
    b = torch.roll(b, shifts=-1, dims=1)
    b[:, -1, :] = messages
    new_buffer[node_ids] = b
    
    new_lengths[node_ids] = torch.clamp(lengths[node_ids] + 1, max=self.seq_len)
    return new_buffer, new_lengths

  def _compute_memory(self, buffer, lengths, node_ids):
    b = buffer[node_ids] # [batch, seq_len, msg_dim]
    l = lengths[node_ids] # [batch]
    
    b = self.message_proj(b) # [batch, seq_len, mem_dim]
    
    batch_size = b.size(0)
    seq_range = torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
    valid_mask = seq_range >= (self.seq_len - l.unsqueeze(1))
    pad_mask = ~valid_mask
    
    # In PyTorch, if a row in pad_mask is all True, attention might return NaN.
    # To prevent this (e.g. l=0), we can unmask the last element artificially.
    is_empty = (l == 0)
    if is_empty.any():
        pad_mask[is_empty, -1] = False
        valid_mask[is_empty, -1] = True
    
    out = self.transformer(b, src_key_padding_mask=pad_mask)
    
    out = out * valid_mask.unsqueeze(-1).float()
    
    sum_out = out.sum(dim=1)
    mem = sum_out / torch.clamp(l.unsqueeze(-1).float(), min=1)
    
    return mem

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    self.memory.last_update[unique_node_ids] = timestamps

    # Update buffer in-place
    b = self.memory.message_buffer[unique_node_ids]
    b = torch.roll(b, shifts=-1, dims=1)
    b[:, -1, :] = unique_messages
    self.memory.message_buffer[unique_node_ids] = b
    
    self.memory.buffer_lengths[unique_node_ids] = torch.clamp(self.memory.buffer_lengths[unique_node_ids] + 1, max=self.seq_len)

    updated_memory = self._compute_memory(self.memory.message_buffer, self.memory.buffer_lengths, unique_node_ids)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    new_buffer, new_lengths = self._append_to_buffer(self.memory.message_buffer, self.memory.buffer_lengths, unique_node_ids, unique_messages)

    computed_memory = self._compute_memory(new_buffer, new_lengths, unique_node_ids)
    
    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = computed_memory

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "transformer":
    return TransformerMemoryUpdater(memory, message_dimension, memory_dimension, device)
