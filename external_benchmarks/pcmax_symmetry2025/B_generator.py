import os                               # Required for interacting with operating system
import numpy as np                      # Required for mathematical functions and working with data
import shutil


problem_instances = 100
random_seed = 1
directory_path="./B"

T = np.array([
 [  4,   7,   9,  11,  13,  16,  18,  22,  27,  33,  40,  48,  59,  72,  86, 104],
 [  9,  11,  14,  17,  19,  22,  25,  29,  34,  40,  47,  56,  67,  79,  94, 111],
 [ 12,  16,  18,  22,  25,  28,  32,  36,  42,  48,  55,  64,  75,  87, 102, 118],
 [ 15,  20,  25,  28,  31,  35,  39,  44,  49,  56,  63,  72,  83,  95, 110, 127],
 [ 18,  24,  29,  33,  38,  42,  46,  52,  57,  64,  72,  81,  92, 104, 118, 135],
 [ 22,  28,  34,  39,  44,  49,  54,  59,  65,  72,  80,  90, 100, 113, 127, 144],
 [ 26,  33,  39,  45,  50,  56,  61,  67,  74,  81,  89,  99, 110, 122, 137, 153],
 [ 29,  37,  44,  51,  57,  63,  69,  75,  82,  90,  99, 108, 119, 132, 146, 163],
 [ 33,  41,  49,  57,  63,  70,  77,  84,  91,  99, 108, 118, 129, 142, 156, 173],
 [ 36,  45,  55,  62,  70,  77,  85,  92, 100, 108, 118, 128, 139, 152, 167, 183],
 [ 39,  50,  59,  68,  77,  85,  93, 101, 109, 118, 127, 138, 150, 163, 177, 194],
 [ 42,  54,  64,  74,  83,  92, 101, 109, 118, 128, 138, 148, 160, 174, 188, 205],
 [ 45,  58,  69,  80,  90, 100, 109, 118, 128, 137, 148, 159, 171, 185, 200, 216],
 [ 48,  62,  74,  86,  97, 107, 117, 127, 137, 147, 158, 170, 182, 196, 211, 228],
 [ 51,  66,  79,  92, 104, 115, 125, 136, 147, 157, 169, 181, 194, 208, 223, 240],
 [ 54,  70,  84,  98, 110, 122, 134, 145, 156, 168, 180, 192, 205, 220, 236, 253],
 [ 56,  73,  89, 103, 117, 130, 142, 154, 166, 178, 190, 203, 217, 232, 248, 265],
 [ 59,  77,  94, 109, 124, 137, 150, 163, 176, 189, 202, 215, 229, 244, 261, 278]])



def getFileName(dirName, value_m, value_n, counter, distribution):
    p = (2 - len(str(counter))) * "0"
    path = directory_path + "/" + dirName + "/" + (2 - len(str(value_m))) * "0" + str(value_m) + "_" + (3 - len(str(value_n))) * "0" + str(value_n) + "_" + distribution + "_" + (2 - len(str(counter))) * "0" + str(counter) + ".dat"
    return path


def generate_B_instances():
    
    total = 0
    for c in range(1,19):
        print(c)

        os.mkdir(directory_path + "/" + str(c))
        for m in range(2,17):
            for i in range(problem_instances):
                data = np.random.randint(1, 10**c, size=T[c-1][m-2])
                distr = str(c)
                path = getFileName(str(c), m, T[c-1][m-2], i, distr)
                
                with open(path, "w", encoding = "UTF-8") as f:
                    p = " ".join(map(str, list(data)))
                    f.writelines([str(m) + " " + str(T[c-1][m-2]) + "\n", p])
               
                total += 1
    
    print("Generating B instances done... Total: " + str(total))
    return total
      

def main():
    
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)  # Remove directory and all contents
    os.mkdir(directory_path)
    
    np.random.seed(random_seed)
    a = generate_B_instances()

    totalNumberOfInstances = a
    print("Total generated: " + str(totalNumberOfInstances) + " instances")
    
    return 0

if __name__ == "__main__":
    main()