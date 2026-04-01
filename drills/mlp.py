import torch
import torch.nn as nn
from typing import List, Optional

class Gated_MLP(nn.Module):
    def __init__(self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        num_layers: int = 2,
        act: nn.Module = nn.GELU,
        dropout_rate: float = 0.0,
        ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.act = act()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        current_in_size = in_size
        for i in range(num_layers):
            # Final layer maps to out_size, others to hidden_size
            current_out_size = out_size if i == num_layers - 1 else hidden_size
            self.layers.append(nn.Linear(current_in_size, current_out_size))
            current_in_size = current_out_size
    
    def forward(self, x: torch.Tensor, masks: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        args:
            x: input tensor (batch_size, in_size)
            masks [optional]: list of tensors, each of size hidden_size
                                length should be num_layers - 1 (each neuron in each hidden layer has a mask)            
        """
        
        if masks is not None and len(masks) != self.num_layers - 1:
            raise ValueError(f"Expected {self.num_layers - 1} masks, got {len(masks)}")

        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply Activation, Dropout, and Masks ONLY if not the last layer
            if i < self.num_layers - 1:
                x = self.act(x)
                x = self.dropout(x)            

            # apply masks to hidden layers ONLY (not output layer)
            if masks is not None:
                if i < self.num_layers - 1:
                    with torch.no_grad():                
                        x = x * masks[i]
        
        return x