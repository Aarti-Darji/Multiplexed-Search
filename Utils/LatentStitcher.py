# #### Libraries

# from os import listdir, remove
# from os.path import join

# import pandas as pd

# from Utils.aux import load_latent_space

# #### Functions and Classes

# class LatentStitcher:
#     def __init__(self, latent_directory):
#         self.latent_directory = latent_directory

    
#     @property
#     def parsed_names(self):
#         parsed_names = {}

#         for name in listdir(self.latent_directory):
            
            
#             if name.endswith(".data"):
#                 #print(name)
#                 _,_,_,_,_,channel_number,coord = name.split("_")
                
#                 name=name.replace('_'+channel_number,'')
#                 name=name.replace('_'+coord,'')
#                 fname=name.replace('pred_','')

#                 coord = coord.split(".data")[0]

#                 if fname in parsed_names.keys():
#                     parsed_names[fname].append(self._str_coord_to_tuple(coord,channel_number))
#                     #parsed_names[fname].append(channel_number)
#                 else:
#                     parsed_names[fname] = [self._str_coord_to_tuple(coord,channel_number)]
#                     #parsed_names[fname] = [channel_number]

#         return parsed_names


#     def stitch(self):
#         #gdc_meta = pd.read_csv("/mnt/mxn2498/projects/uta_cancer_search/Datasets/clam_test_metadata.csv")
#         results = pd.DataFrame(columns=["filename", "sampled_coords", "channel_number", "latent_value"])
        
#         #print(self.parsed_names.items())
        
#         for fname, coords in self.parsed_names.items():
#             #meta = gdc_meta[gdc_meta["filename"] == fname + ".svs"]
#             #primary_site = meta["primary_site"].to_list()[0]
#             #primary_site='Colon'

#             for coord in coords:
#                 latent = load_latent_space(join(self.latent_directory, f"pred_{fname}_{coord[2]}_({coord[0]},{coord[1]}).data"))
#                 results.loc[len(results.index)] = [fname, coord[:2],coord[2], latent.cpu()]
        
#             self._clean_directory(fname)
        
#         results.to_csv(join(self.latent_directory, "latent_spaces.csv"), index=False)


#     def _str_coord_to_tuple(self, str_coord,channel_number):
#         coord = str_coord.strip(")(").split(",")
#         coord_list = [int(c) for c in coord]
        
#         coord_list=tuple(coord_list)

        


#         new_tuple = coord_list + (channel_number,)

        
        
#         return new_tuple

    
#     def _clean_directory(self, fname):
#         for name in listdir(self.latent_directory):
#             if name.startswith(f"pred_{fname}") and name.endswith(".data"):

#                 remove(join(self.latent_directory, name))


from os import listdir, remove
from os.path import join
import pandas as pd
import torch
import numpy as np
from Utils.aux import load_latent_space

class LatentStitcher:
    def __init__(self, latent_directory):
        self.latent_directory = latent_directory

    @property
    def parsed_names(self):
        parsed_names = {}
        for name in listdir(self.latent_directory):
            if name.endswith(".data"):
                _, _, _, _, _, channel_number, coord = name.split("_")
                name = name.replace('_' + channel_number, '').replace('_' + coord, '')
                fname = name.replace('pred_', '')
                coord = coord.split(".data")[0]
                if fname in parsed_names.keys():
                    parsed_names[fname].append(self._str_coord_to_tuple(coord, channel_number))
                else:
                    parsed_names[fname] = [self._str_coord_to_tuple(coord, channel_number)]
        return parsed_names

    def stitch(self):
        results = pd.DataFrame(columns=["filename", "latent_matrix"])
        for fname, coords in self.parsed_names.items():
            channel_data = {}
            for coord in coords:
                channel_number = coord[2]
                latent_vector = load_latent_space(join(self.latent_directory, f"pred_{fname}_{channel_number}_({coord[0]},{coord[1]}).data"))
                if channel_number not in channel_data:
                    channel_data[channel_number] = []
                channel_data[channel_number].append(latent_vector.cpu().numpy())  # Loaded as NumPy array
        
                image_matrix = []
                for channel, vectors in channel_data.items():
                    if len(vectors) > 0:
                        tensor_vectors = [torch.tensor(vector) for vector in vectors]
                        image_matrix.append(torch.stack(tensor_vectors))

                matrix_str = self._matrix_to_string(torch.stack(image_matrix))
        
        # Add a new row to the DataFrame
                index = len(results)
                results.loc[index, "filename"] = fname
                results.loc[index, "latent_matrix"] = matrix_str

            results.to_csv(join(self.latent_directory, "latent_matrices.csv"), index=False)



    def _matrix_to_string(self, matrix):
        np_matrix = matrix.numpy()
        matrix_str = np.array2string(np_matrix, separator=',', max_line_width=np.inf)
        return matrix_str

    def _str_coord_to_tuple(self, str_coord, channel_number):
        coord = str_coord.strip(")(").split(",")
        coord_list = tuple([int(c) for c in coord] + [channel_number])
        return coord_list

    def _clean_directory(self, fname):
        for name in listdir(self.latent_directory):
            if name.startswith(f"pred_{fname}") and name.endswith(".data"):
                remove(join(self.latent_directory, name))

