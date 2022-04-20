using MLNet.Regression.LinearRegression;
using Numpy;

namespace MLNet.Regression
{
    /// <summary>
    ///     PolynomialFeatures
    ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
    /// </summary>
    public class PolynomialFeatures : MultipleLinearRegression
    {
        public PolynomialFeatures(
            Constraint constraint = Constraint.None,
            int degree = 1,
            double alhpa = 0.3)
            : base("PolynomialFeatures")
        {
            Constraint = constraint;
            Degree = degree;
            Alpha = alhpa;
        }

        public int Degree { set; get; }
        public double Alpha { set; get; }

        public Constraint Constraint { set; get; }


        /// <summary>
        ///     x => 1,x,x^2,x^3,...,x^N
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray convert(NDarray x)
        {
            var batch = x.shape[0];
            var features = x.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(x);
            var npX = np.ones(Degree + 1, batch);
            Enumerable.Range(1, Degree).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }


        /// <summary>
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public NDarray SloveNone(NDarray x, NDarray y)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x));
            var res = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
            return res;
        }


        /// <summary>
        ///     L1 Solve Lasso Todo
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public NDarray SloveL1(NDarray x, NDarray y)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x) + Alpha * np.eye(Degree + 1));
            var res = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
            return res;
        }

        /// <summary>
        ///     L2 Solve Ridge
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public NDarray SloveL2(NDarray x, NDarray y)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x) + Alpha * np.eye(Degree + 1));
            var res = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
            return res;
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