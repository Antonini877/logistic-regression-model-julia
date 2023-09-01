include("logistic_regression.jl")
using .LogisticRegression

function main()
    
    X_train = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    y_train = [0, 1, 1]
    
    learning_rate = 0.01
    num_iterations = 1000

    # Initialize parameters
    model = LogisticRegression.initialize_parameters(
        size(X_train, 2),
        learning_rate
        )


    # Train the model
    model = LogisticRegression.train!(X_train, y_train, model, num_iterations)
    
    X_predict = [3.4 2.2 7.1]
    
    println(LogisticRegression.predict(X_predict, model))

end

main()