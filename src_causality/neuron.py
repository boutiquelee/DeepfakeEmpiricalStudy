import os
import re
import chardet

results_folder = "../results/neuron/"
file_list = os.listdir(results_folder)

results_dict = {}

for file_name in file_list:
    with open(results_folder + file_name, "rb") as f:
        byte_content = f.read()
        result = chardet.detect(byte_content)
        file_encoding = result['encoding']
        lines = byte_content.decode(file_encoding).split('\n')
        for line in lines:
            num = int(re.findall("\d+", line.strip())[0])
            if num not in results_dict:
                results_dict[num] = 1
            else:
                results_dict[num] += 1

for i in range(1, 6):
    count = sum([1 for v in results_dict.values() if v == i])
    print(f"the number of {i}-time identified neurons is {count}")
