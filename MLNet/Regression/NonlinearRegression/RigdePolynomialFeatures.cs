using Numpy;

namespace MLNet.Regression
{
    public class RigdePolynomialFeatures : PolynomialFeatures
    {
        public RigdePolynomialFeatures(int degree,
            double alpha) : base(degree)
        {
            Alpha = alpha;
        }

        public double Alpha { set; get; }

        internal override void fit(NDarray x, NDarray y, double learning_rate, int epoch)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x) + Alpha * np.eye(Degree + 1));
            Resolve = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
        }
    }
}