using MLNet.Utils;
using Numpy;

namespace MLNet.Regression.LinearRegression
{
    public class OrdinaryLeastSquares : LinearRegression
    {
        public OrdinaryLeastSquares(
            string name,
            PrimaryType primaryType = PrimaryType.Polynomial,
            int alpha = 16) :
            base(name, primaryType, alpha)
        {
        }

        internal override NDarray fit(NDarray X, NDarray Y)
        {
            throw new NotImplementedException();
        }
    }
}