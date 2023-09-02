module LogisticRegression

    using Random
    using LinearAlgebra
    using DataFrames

    export predict, initialize_parameters, train
   

    mutable struct LogisticRegressionModel
        θ::Array
        bias::Float64
        α::Float64
    end


    σ(x) = 1 / (1 + exp(-x)) # Sigmoid function

    function predict(X::Array, model::LogisticRegressionModel)
        z = (X * model.θ).+ model.bias
        y_pred = σ.(z)
        return y_pred
    end

    function initialize_parameters(num_features::Int, learning_rate::Float64)
        return LogisticRegressionModel(
            zeros(num_features),
            0.0,
            learning_rate
            )
    end

    function forward(X::Matrix, θ::Array, b)
        z = (X * θ).+ b
        y_pred = σ.(z)
        return y_pred
    end

    function train!(X::Matrix, y::Array, model::LogisticRegressionModel, num_iterations::Int)
        return backpropagation!(X, y, model, num_iterations)
    end

    function train!(X::DataFrame, y::Array, model::LogisticRegressionModel, num_iterations::Int)
        X = Matrix(X)
        return backpropagation!(X, y, model, num_iterations)
    end

    function train!(X::DataFrame, y::DataFrame, model::LogisticRegressionModel, num_iterations::Int)
        X = Matrix(X)
        y = Array(y)
        return backpropagation!(X, y, model, num_iterations)
    end

    function backpropagation!(X::Matrix, y::Array, model::LogisticRegressionModel, num_iterations::Int)
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