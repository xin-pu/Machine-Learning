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

        /// <summary>
        ///     X/Y
        ///     np.linalg.lstsq(A, B)
        ///     np.linalg.pinv(A).dot(A)
        /// </summary>
        /// <param name="X"></param>
        /// <param name="Y"></param>
        /// <returns></returns>
        internal override NDarray fit(NDarray X, NDarray Y)
        {
            return np.linalg.pinv(X).dot(Y);
        }
    }
}