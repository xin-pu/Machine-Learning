using MathNet.Numerics.LinearAlgebra.Double;

namespace MLNet.Regression.LinearRegression;

public class RidgeRegression : LinearRegression
{
    public override void Fit(Matrix X, Matrix Y)
    {
        Theta = (Matrix)X.Svd().Solve(Y);
    }
}