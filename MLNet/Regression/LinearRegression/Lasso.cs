namespace MLNet.Regression.LinearRegression
{
    public class Lasso : PolynomialFeatures
    {
        public Lasso(int degree = 5, double alpha = 0.3)
            : base(Constraint.L1, degree, alpha)
        {
        }
    }
}