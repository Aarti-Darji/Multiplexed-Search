#### Libraries

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


# import json
# import os
# from os.path import join
# from Utils.aux import load_latent_space

# class LatentStitcher:
#     def __init__(self, latent_directory):
#         self.latent_directory = latent_directory

#     @property
#     def parsed_names(self):
#         parsed_names = {}
#         for name in os.listdir(self.latent_directory):
#             if name.endswith(".data"):
#                 _, _, _, _, _, channel_number, coord = name.split("_")
#                 name = name.replace('_' + channel_number, '')
#                 name = name.replace('_' + coord, '')
#                 fname = name.replace('pred_', '')
#                 coord = coord.split(".data")[0]
#                 if fname in parsed_names:
#                     parsed_names[fname].append(self._str_coord_to_tuple(coord, channel_number))
#                 else:
#                     parsed_names[fname] = [self._str_coord_to_tuple(coord, channel_number)]
#         return parsed_names

#     def stitch(self):
#         all_images_data = {}
#         for fname, coords in self.parsed_names.items():
#             image_matrix = {}
#             for coord in coords:
#                 channel, patch_x, patch_y = coord
#                 latent_file_path = join(self.latent_directory, f"pred_{fname}_{channel}_({patch_x},{patch_y}).data")
#                 if not os.path.exists(latent_file_path):
#                     print(f"Warning: File not found {latent_file_path}")
#                     continue
#                 latent = load_latent_space(latent_file_path)
#                 if channel not in image_matrix:
#                     image_matrix[channel] = []
#                 image_matrix[channel].append({"patch_x": patch_x, "patch_y": patch_y, "values": latent.numpy().tolist()})
            
#             all_images_data[fname] = image_matrix

#         with open(join(self.latent_directory, "latent_matrices.json"), 'w') as outfile:
#             json.dump(all_images_data, outfile, indent=4)

#     def _str_coord_to_tuple(self, str_coord, channel_number):
#         coord = str_coord.strip(")(").split(",")
#         coord_list = [int(c) for c in coord]
#         channel_number = int(channel_number)
#         return channel_number, coord_list[0], coord_list[1]

#     def _clean_directory(self, fname):
#         for name in os.listdir(self.latent_directory):
#             if name.startswith(f"pred_{fname}") and name.endswith(".data"):
#                 os.remove(join(self.latent_directory, name))

import numpy as np
from os import listdir, remove
from os.path import join
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
                name = name.replace('_' + channel_number, '')
                name = name.replace('_' + coord, '')
                fname = name.replace('pred_', '')
                coord = coord.split(".data")[0]
                if fname in parsed_names:
                    parsed_names[fname].append(self._str_coord_to_tuple(coord, channel_number))
                else:
                    parsed_names[fname] = [self._str_coord_to_tuple(coord, channel_number)]
        return parsed_names

    def stitch(self):
        for fname, coords in self.parsed_names.items():
            # Determine the shape of the matrix
            num_channels = max([int(coord[2]) for coord in coords]) + 1
            num_patches = len(coords)
            latent_dim = len(load_latent_space(join(self.latent_directory, f"pred_{fname}_{coords[0][2]}_({coords[0][0]},{coords[0][1]}).data")))

            # Initialize matrix
            matrix = np.zeros((num_channels, num_patches, latent_dim))

            for i, coord in enumerate(coords):
                latent = load_latent_space(join(self.latent_directory, f"pred_{fname}_{coord[2]}_({coord[0]},{coord[1]}).data"))
                matrix[int(coord[2]), i, :] = latent

            # Save the matrix
            np.save(join(self.latent_directory, f"{fname}_latent_matrix.npy"), matrix)

            # Clean up directory
            self._clean_directory(fname)

    def _str_coord_to_tuple(self, str_coord, channel_number):
        coord = str_coord.strip(")(").split(",")
        return tuple([int(c) for c in coord] + [channel_number])

    def _clean_directory(self, fname):
        for name in listdir(self.latent_directory):
            if name.startswith(f"pred_{fname}") and name.endswith(".data"):
                remove(join(self.latent_directory, name))