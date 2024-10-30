inputs = [1,2,3,2.5]
weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.2,0.8,-0.5,1.0]
weights3 = [0.2,0.8,-0.5,1.0]
bias = 2.0


output = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + (inputs[3] * weights[3])+ bias

if __name__ == '__main__':
    print(output)