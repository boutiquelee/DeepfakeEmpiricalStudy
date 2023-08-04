import os
from skimage.metrics import structural_similarity as ssim
from skimage import io

dataset_names = ["CELEB","FS", "NT", "DF", "DFD"]

for dataset_name in dataset_names:
    print(f"Dataset: {dataset_name}")

    model_name = "EfficientNet"
    print(model_name)
    model_lists = ["EfficientNet1", "EfficientNet2", "EfficientNet3"]

    folders = [f"../heatset/{model_name}/{model_list}/{dataset_name}" for model_list in model_lists]

    intersection = set(os.listdir(folders[0]))
    for folder in folders[1:]:
        intersection = intersection.intersection(set(os.listdir(folder)))

    sim_dicts = []
    for i in range(len(folders)):
        for j in range(i+1, len(folders)):
            sim_dict = {}
            sim_dicts.append(sim_dict)

    for filename in intersection:
        img = []
        for folder in folders:
            img.append(io.imread(os.path.join(folder, filename)))

        k = 0
        for i in range(len(folders)):
            for j in range(i+1, len(folders)):
                sim = ssim(img[i], img[j], multichannel=True)
                sim_dicts[k][filename] = sim
                k += 1

    for filename in intersection:
        avg = sum([sim_dict[filename] for sim_dict in sim_dicts]) / len(sim_dicts)
        
    count_less_07 = 0
    count_07_to_09 = 0
    count_greater_09 = 0

    for filename in intersection:
        avg = sum([sim_dict[filename] for sim_dict in sim_dicts]) / len(sim_dicts)
        if avg < 0.7:
            count_less_07 += 1
        elif avg < 0.9:
            count_07_to_09 += 1
        else:
            count_greater_09 += 1

    print(f"Number of files with average SSIM < 0.7: {count_less_07}")
    print(f"Number of files with average SSIM between 0.7 and 0.9: {count_07_to_09}")
    print(f"Number of files with average SSIM > 0.9: {count_greater_09}")
