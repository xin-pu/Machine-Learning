using MathNet.Numerics.LinearAlgebra.Double;
using MLNet.Model;

namespace MLNet.Regression.LinearRegression;

public abstract class LinearRegression : MLModel
{
    protected LinearRegression()
        : base("LinearRegression")
    {
    }

    public Matrix Theta { set; get; }

    public override double Evaluate(Matrix X, Matrix Y)
    {
        throw new NotImplementedException();
    }

    public override Matrix Predict(Matrix X)
    {
        var yPred = Theta.Multiply(X);
        return (Matrix)yPred;
    }

    public override void Save(string path)
    {
        throw new NotImplementedException();
    }

    public override MLModel Load(string path)
    {
        throw new NotImplementedException();
    }
}