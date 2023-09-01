module LogisticRegression

    using Random
    using LinearAlgebra

    export predict, initialize_parameters, train
   

    mutable struct LogisticRegressionModel
        θ::Array
        bias::Float64
        α::Float64
    end


    σ(x) = 1 / (1 + exp(-x)) # Sigmoid function

    function predict(X, model)
        z = (X * model.θ).+ model.bias
        y_pred = σ.(z)
        return y_pred
    end

    function initialize_parameters(num_features, learning_rate)
        return LogisticRegressionModel(
            zeros(num_features),
            0.0,
            learning_rate
            )
    end

    function forward(X, θ, b)
        z = (X * θ).+ b
        y_pred = σ.(z)
        return y_pred
    end

    function train!(X, y, model, num_iterations)
        m = length(y)
        for iteration in 1:num_iterations
            y_pred = forward(X, model.θ, model.bias)
            gradient_θ = (X' * (y_pred - y)) / m
            gradient_b = sum(y_pred - y) / m
            
            model.θ -= model.α * gradient_θ
            model.bias -= model.α * gradient_b
        end
        return model
    end

   

end