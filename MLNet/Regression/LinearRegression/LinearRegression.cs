using Numpy;

namespace MLNet.Regression.LinearRegression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public class LMS : AbstractLinearRegression
    {
        public LMS()
            : base("LMS")
        {
        }


        internal override NDarray slove(NDarray x, NDarray y)
        {
            return np.linalg.pinv(x).dot(y);
        }
    }
}