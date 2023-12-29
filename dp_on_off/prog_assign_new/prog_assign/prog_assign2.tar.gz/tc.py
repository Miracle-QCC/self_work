import math

import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method

        self.offset = []
        self.tiles = []
        self.tile_width = tile_width
        self.num_tilings = num_tilings
        for i in range(len(self.tile_width)):
            self.tiles.append((math.ceil((state_high[i] - state_low[i]) / tile_width[i])+1))
        for i in range(self.num_tilings):
            self.offset.append((state_low - (i / num_tilings) * tile_width))
        self.weight = np.zeros(np.append(self.num_tilings, self.tiles))


    def __call__(self,s):
        # TODO: implement this method
        tiles = self.get_tiles(s)
        res = 0
        for tile,weight in zip(tiles, self.weight):
            res += weight[tile[0], tile[1]]
        return res

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        tiles_tau = self.get_tiles(s_tau)
        for idx,tile in enumerate(tiles_tau):
            self.weight[idx,tile[0],tile[1]] += alpha * (G - self(s_tau))

    def get_tiles(self, state):
        """
        Get the tiles for a given state.
        """
        tiles = []
        shape = self.tiles
        for i in range(self.num_tilings):
            tile = []
            for j in range(len(state)):
                tile.append(int((state[j] - self.offset[i][j]) / self.tile_width[j]))
            tiles.append(tile)
        return tuple(tiles)
