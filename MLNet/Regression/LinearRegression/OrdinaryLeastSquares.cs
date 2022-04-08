using MathNet.Numerics.LinearAlgebra.Double;

namespace MLNet.Regression.LinearRegression;

internal class OrdinaryLeastSquares : LinearRegression
{
    public override void Fit(Matrix X, Matrix Y)
    {
        Theta = (Matrix)X.Svd().Solve(Y);
    }
}