using MLNet.Utils;
using Numpy;

namespace MLNet.Regression.LinearRegression
{
    public class OrdinaryLeastSquares : LinearRegression
    {
        public OrdinaryLeastSquares(
            PrimaryType primaryType = PrimaryType.Polynomial,
            int alpha = 16) :
            base("OrdinaryLeastSquares", primaryType, alpha)
        {
        }

        internal override NDarray slove(NDarray X, NDarray Y)
        {
            return np.linalg.pinv(X).dot(Y);
        }

        internal override NDarray fit(NDarray X, NDarray Y)
        {
            throw new NotImplementedException();
        }
    }
}