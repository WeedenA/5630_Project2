def main():
    args = parse_arguments()
    data = Dataset(args.dataset_dir)
    class_weights = []

    for i in range(10):
        if args.kernel:
            print('Using RBF kernel')
            dataset = data.sepData(i)
            class_weights.append(kernelized_pegasos(
                x=dataset['data'],
                y=dataset['labels'],
                kernel=kernel_function,
                iterations=args.iterations
            ))
        else:
            dataset = data.sepData(i)
            class_weights.append(pegasos(
                x=dataset['data'],
                y=dataset['labels'],
                iterations=args.iterations
            ))

    # Testing
    errors = 0
    for i in range(len(data.ytest)):
        predictions = []
        for k in range(10):
            weights = class_weights[k]
            if args.kernel:
                decision = 0
                for j in range(len(data.ytrain)):
                    decision += weights[j] * data.ytrain[j] * kernel_function(data.xtrain[j], data.xtest[i])
            else:
                decision = weights @ data.xtest[i].T
            predictions.append(decision)
        predictions = np.array(predictions)
        class_label = predictions.argmax()
        if class_label != data.ytest[i]: errors += 1
    accuracy = 1 - errors / len(data.ytest)
    print('Error:', errors / len(data.ytest))
    print('Accuracy:', accuracy)


main()