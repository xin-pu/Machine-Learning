using MLNet.LearningModel;
using MLNet.Utils;
using Numpy;

namespace MLNet.Regression.LinearRegression
{
    public class LinearRegression : LinearModel
    {
        public LinearRegression(
            PrimaryType primaryType,
            int alpha)
            : base("LinearRegression", primaryType, alpha)
        {
        }


        internal override NDarray fit(NDarray X, NDarray Y)
        {
            return np.linalg.solve(X, Y);
        }
    }
}