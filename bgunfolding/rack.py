import msgpack
import msgpack_numpy as m
import os
import numpy as np

class Rack():
    
    ''' Gather data easily and store, write and read nested dictionaries. '''
    
    def __init__(self):
        self.g = dict() # gather dict 
        self.s = dict() # storage dict
        self.locs = []
    
    def gather(self, **kwargs):
        ''' Store data into gathering dict `g`. Items can be added inside a loop
        and will be appended to the corresponding key. See Example.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments
            
        Example
        -------
        >> r = Store()
        >> r.gather(x = 5)
        >> for i in range(5):
        >>     r.gather(i = i)
        >>
        >> print(r.g)
        >> {'x': 5, 'i': [0, 1, 2, 3, 4]}
        '''
        
        for key, val in kwargs.items():
            if (str(key) in self.g.keys()) == True:
                x = self.g[key]
                if type(x) != list:
                    self.g[key] = [x] + [val]
                else:
                    x.append(val)
                    self.g[key] = x

            else:
                self.g.update({key: val})
            
    def store(self, loc):
        '''
        Stores gather dictionary `g` into storage dictionary `s` at given location.
        
        Parameters
        ----------
        loc : str
            Location (e.g. /data/sample_0/)
        '''
        locs = loc.split('/')
        self.locs.append(loc)
        while '' in locs:
            locs.remove('')
            
        if len(locs) == 0:
            self.s.update(self.g)
            
        else:
            d = self.nest(locs)
            d = self.change_key(d, locs[-1], self.g, locs)

            if d == self.s:
                pass
            else:
                try:
                    self.s = self.merge_dicts(self.s, d)
                    self.g = dict()
                except:
                    pass
                
    def nest(self, locs):
        ''' Create a nested dictionary 
        
        Parameters
        ----------
        locs : list
        '''
        if len(locs) == 0:
            return locs
        else:
            x = locs[0]
            d = {x : self.nest(locs[1:])}
            
            return d

    def change_key(self, d, required_key, new_value, locs):
        ''' Changes key of nested dictionary at location. '''
        i = 0
        count = locs.count(required_key)
        for k, v in d.items():
            if isinstance(v, dict):
                self.change_key(v, required_key, new_value, locs[1:])
            if k == required_key:
                i += 1
                if i < count:
                    self.change_key(v, required_key, new_value, locs[1:])
                else:
                    d[k] = new_value
        
        return d
            
    def write(self, fp, overwrite = False):
        ''' Write storage dictionray `s` to binary file. '''
        m.patch()
        
        self.gather(locs = np.unique(self.locs))
        self.store('/')
        
        if not os.path.exists(fp):
            print(f'\'{fp}\' saved')
            binary = msgpack.packb(self.s, use_bin_type  = True)
            with open(fp, 'wb') as file:
                file.write(binary)

        elif overwrite:
            print(f'\'{fp}\' saved (overwritten)')
            binary = msgpack.packb(self.s, use_bin_type  = True)
            with open(fp, 'wb') as file:
                file.write(binary)
                
        else:
            print(f'\'{fp}\' already exists. Set `overwrite = True` to overwrite.')

    def read(self, fp):
        ''' Read binary file. Uses `msgpack` and `mspack_numpy`. '''
        
        m.patch()
        with open(fp, 'rb') as file:
            rec = msgpack.unpackb(file.read(), encoding = 'utf-8')
        self.s = rec
        
    def tree(self):
        ''' List all locations. '''
        
        for l in self.locs:
            print(l)
        
    def get(self):
        ''' Returns storage dictionary `s`. '''
        
        return self.s

    def merge_dicts(self, d1, d2):
        ''' Recursively merges d2 into d1 '''
        
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return d2
        for k in d2:
            if k in d1:
                d1[k] = self.merge_dicts(d1[k], d2[k])
            else:
                d1[k] = d2[k]
        
        return d1
    
    def gather_dict(self, d):
        ''' Gather a given dict `d`. '''
        
        for key, val in d.items():
            if (str(key) in self.g.keys()) == True:
                x = self.g[key]
                if type(x) != list:
                    self.g[key] = [x] + [val]
                else:
                    x.append(val)
                    self.g[key] = x

            else:
                self.g.update({key: val})
        
    def reset(self, d):
        self.g = dict()
        self.s = dict()
        self.locs = []
        
    def load(self, loc):
        locs = loc.split('/')
        self.locs.append(loc)
        while '' in locs:
            locs.remove('')
        
        return self._get_nested(self.s, locs)
        
    def _get_nested(self, d, path):
        if len(path) == 0:
            return d

        cur, path = path[0], path[1:]
        return self._get_nested(d[cur], path) # recursion