using MLNet.LearningModel;
using Numpy;

namespace MLNet.Regression.LinearRegression
{
    /// <summary>
    ///     PolynomialFeatures
    ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
    /// </summary>
    public class PolynomialFeatures : LinearModel
    {
        public PolynomialFeatures(int degree = 1)
            : base("PolynomialFeatures")
        {
            Degree = degree;
        }

        public int Degree { set; get; }

        public override NDarray Slove(NDarray x, NDarray y)
        {
            if (x.ndim != 2 || x.shape[1] != 1)
                throw new ArgumentException("Input X Shape should be [n,1]");

            var xTranspose = np.transpose(x);
            var npX = np.ones(Degree + 1, x.shape[0]);
            Enumerable.Range(0, Degree + 1).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return np.linalg.pinv(npX).dot(y);
        }

        public override NDarray SGD(NDarray x, NDarray y)
        {
            throw new NotImplementedException();
        }

        public override NDarray Pred(NDarray x)
        {
            return np.matmul(CvtToPoly(x), Theda);
        }

        internal NDarray CvtToPoly(NDarray x)
        {
            var xTranspose = np.transpose(x);
            var npX = np.ones(Degree + 1, x.shape[0]);
            Enumerable.Range(0, Degree + 1).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}