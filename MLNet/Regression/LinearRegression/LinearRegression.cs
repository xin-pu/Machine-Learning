using MLNet.LearningModel;
using Numpy;

namespace MLNet.Regression.LinearRegression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public class LinearRegression : LinearModel
    {
        public LinearRegression()
            : base("LinearRegression")
        {
        }


        public override NDarray Slove(NDarray x, NDarray y)
        {
            return np.linalg.pinv(x).dot(y);
        }

        public override NDarray SGD(NDarray x, NDarray y)
        {
            throw new NotImplementedException();
        }

        public override NDarray Pred(NDarray x)
        {
            return x.multiply(TheDa);
        }
    }
}