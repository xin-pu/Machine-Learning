using MathNet.Numerics.LinearAlgebra.Double;
using MLNet.Model;

namespace MLNet.Regression;

public abstract class LinearRegression : MLModel
{
    protected LinearRegression(Matrix slove)
        : base("LinearRegression")
    {
        Slove = slove;
    }

    public Matrix Slove { set; get; }
}