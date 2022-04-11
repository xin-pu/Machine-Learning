namespace MLNet.Regression.LinearRegression
{
    /// <summary>
    ///     岭回归
    /// </summary>
    public class Ridge : PolynomialFeatures
    {
        public Ridge(int degree = 5, double alpha = 0.3)
            : base(Constraint.L2, degree, alpha)
        {
        }
    }
}