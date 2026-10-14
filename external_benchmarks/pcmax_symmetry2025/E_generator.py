import os                               # Required for interacting with operating system
import numpy as np                      # Required for mathematical functions and working with data
import shutil


problem_instances = 50
random_seed = 1
directory_path="./E"


def getFileName(dirName, value_m, value_n, counter, distribution):
    p = (2 - len(str(counter))) * "0"
    path = directory_path + "/" + dirName + "/" + (2 - len(str(value_m))) * "0" + str(value_m) + "_" + (3 - len(str(value_n))) * "0" + str(value_n) + "_" + distribution + "_" + (2 - len(str(counter))) * "0" + str(counter) + ".dat"
    return path


def generate_E1_instances():
    option_m = [3,4,5]
    multiplier = [2,3,5]
    distribution = [(1,21), (20,51)]
    
    print("E1:")

    total = 0
    
    os.mkdir(directory_path + "/E1")
    for m in option_m:
        for mul in multiplier:
            n = m * mul
            for d in distribution:
                counter = 0
                for i in range(problem_instances):
                    data = np.random.randint(low=d[0], high=d[1], size=n)
                    distr = "U_" + str(d[0]) + "_" + str(d[1]-1)
                    path = getFileName("E1", m, n, i, distr)
                    
                    with open(path, "w", encoding = "UTF-8") as f:
                        p = " ".join(map(str, list(data)))
                        f.writelines([str(m) + " " + str(n) + "\n", p])
                    counter += 1
                
                total += counter
        
    print("Generating E1 instances done... Total: " + str(total))
    return total


def generate_E2_instances():
    option_m = [2,3,4,6,8,10]
    option_n = [10,30,50,100]
    distribution = [(100,801)]
    
    print("E2:")

    total = 0
    
    os.mkdir(directory_path + "/E2")
    for m in option_m:
        for n in option_n:
            for d in distribution:
                
                if n == 10 and m > 3:
                    continue
                
                counter = 0
                for i in range(problem_instances):
                    data = np.random.randint(low=d[0], high=d[1], size=n)
                    distr = "U_" + str(d[0]) + "_" + str(d[1]-1)
                    path = getFileName("E2", m, n, i, distr)
                    
                    with open(path, "w", encoding = "UTF-8") as f:
                        p = " ".join(map(str, list(data)))
                        f.writelines([str(m) + " " + str(n) + "\n", p])
                    counter += 1
                
                total += counter
        
    print("Generating E2 instances done... Total: " + str(total))
    return total


def generate_E3_instances():
    option_m = [3,5,8,10]
    options = list()
    distribution = [(1,101), (100,201), (100,801)]
    
    for m in option_m:
        options.append((m, 3*m + 1))
        options.append((m, 3*m + 2))
        options.append((m, 4*m + 1))
        options.append((m, 4*m + 2))
        options.append((m, 5*m + 1))
        options.append((m, 5*m + 2))
    
    print("E3:")

    total = 0
    
    os.mkdir(directory_path + "/E3")
    for opt in options:
        m = opt[0]
        n = opt[1]

        for d in distribution:
            counter = 0
            for i in range(problem_instances):
                data = np.random.randint(low=d[0], high=d[1], size=n)
                distr = "U_" + str(d[0]) + "_" + str(d[1]-1)
                path = getFileName("E3", m, n, i, distr)
                
                with open(path, "w", encoding = "UTF-8") as f:
                    p = " ".join(map(str, list(data)))
                    f.writelines([str(m) + " " + str(n) + "\n", p])
                counter += 1
    
            total += counter
        
    print("Generating E3 instances done... Total: " + str(total))
    return total
    

def generate_E4_instances():
    options = [(2,9),(3,10)]
    distribution = [(1,21), (20,51), (50,101), (100,201), (100,801)]
    
    print("E4:")

    total = 0
    
    os.mkdir(directory_path + "/E4")
    for opt in options:
        m = opt[0]
        n = opt[1]
        
        for d in distribution:
            counter = 0            
            for i in range(problem_instances):
                data = np.random.randint(low=d[0], high=d[1], size=n)
                distr = "U_" + str(d[0]) + "_" + str(d[1]-1)
                path = getFileName("E4", m, n, i, distr)

                with open(path, "w", encoding = "UTF-8") as f:
                    p = " ".join(map(str, list(data)))
                    f.writelines([str(m) + " " + str(n) + "\n", p])
                counter += 1

            total += counter
        
    print("Generating E4 instances done... Total: " + str(total))
    return total
                

def main():
    
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)  # Remove directory and all contents
    os.mkdir(directory_path)
    
    np.random.seed(random_seed)
    a = generate_E1_instances()
    b = generate_E2_instances()
    c = generate_E3_instances()
    d = generate_E4_instances()
    
    totalNumberOfInstances = a + b + c + d
    print("Total generated: " + str(totalNumberOfInstances) + " instances")
    
    return 0

if __name__ == "__main__":
    main()