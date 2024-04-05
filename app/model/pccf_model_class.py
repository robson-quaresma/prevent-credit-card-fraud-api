import torch
import torch.nn as nn
import torch.nn.functional as F


class PCCFModelSimple(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # input
        self.input = nn.Linear(7, 20)

        self.h1 = nn.Linear(20,30)
        self.h2 = nn.Linear(30,20)
        self.h3 = nn.Linear(20,30)

        # output
        self.output = nn.Linear(30, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # input
        x = F.relu( self.input(x) )
        x = F.relu( self.h1(x) )
        x = F.relu( self.h2(x) )
        x = F.relu( self.h3(x) )

        # output
        return self.output(x)

