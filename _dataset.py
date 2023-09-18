import torch.utils.data as tdata
import numpy as np
import math

class Seismic(tdata.Dataset):
    def __init__(self, 
                 data, 
                 azimuths,
                 signal_start=50,
                 signal_length = 200,
                 label_sigma = 10,
                 is_aug_shift =False,
                 aug_shift_range =[-10,10],
   
                 is_minus = True, 
                 label_sep = 1
                 ):
        
        self.label_sigma = label_sigma
        self.label_sep = label_sep
        self.signal_start = signal_start
        self.signal_length = signal_length
        
        self.is_aug_shift = is_aug_shift
        self.aug_shift_range = aug_shift_range
        
        self.is_minus= is_minus
        self.azimuths = azimuths
        self.azimuths_probs = np.array(list(map(self.y_v, self.azimuths)))
            
        self.data_cached = data
        if not self.is_aug_shift:
            self.data_cached = self._data_normalize(self.data_cached[:,:,self.signal_start:self.signal_start+self.signal_length], batch=True)

        
    def __len__(self):
        return self.azimuths.shape[0]
    
    def __getitem__(self, index):
        if not self.is_aug_shift:
            x = self.data_cached[index]
        else:
            rand = np.random.randint(self.signal_start + self.aug_shift_range[0], self.signal_start+self.aug_shift_range[1])
            x = self.data_cached[index,:,rand:rand+self.signal_length]
            x = self._data_normalize(x, batch=False)
     
        y = self.azimuths_probs[index]
        return x.astype(np.float32), y.astype(np.float32)

    def _data_normalize(self, x, batch=True):
        if batch:
            t_min = np.min(x, axis=(1,2), keepdims=True)
            t_max = np.max(x, axis=(1,2), keepdims=True)
            if self.is_minus:
                return (x - t_min) / (t_max-t_min)
            else:
                return x / (t_max - t_min)
        else:
            if self.is_minus:
                return (x - x.min()) / (x.max() - x.min())
            else:
                return x / (x.max() - x.min())
    
      
    def y_v(self, v):
        # transform azimuth value to a probability distribution.
        sigma, sep = self.label_sigma, self.label_sep
        i = np.arange(0,360, sep)
        if sigma == 0:
            return np.where(i == round(v), 1.0, 0.0)
        d = np.min(np.concatenate([
                    np.abs(v-i).reshape(1,-1), 
                    np.abs(v+360-i).reshape(1,-1), 
                    np.abs(v-360-i).reshape(1,-1)]),
                axis=0)
        return 1/np.sqrt(2*np.pi)/sigma * np.exp(-d**2/2/(sigma**2)) * sep


    def v_y(self, y, hard=False):
        # transform a probability distribution to azimuth value.
        if self.label_sigma == 0 or hard:
            return np.argmax(y)
        sep = self.label_sep
        i = np.arange(0, 360, sep)
        index = np.arange(0,int(360/sep)).astype(np.int32)
        max_i = i[np.argmax(y)]
        if abs(max_i - 180) > 100:
            i = np.arange(-180, 180, sep)
            index = np.arange(int(-180/sep), int(180/sep)).astype(np.int32)
        return np.sum(y[index] * i) % 360 %360
    
    
