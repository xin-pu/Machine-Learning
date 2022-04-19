using MLNet.LearningModel;
using MLNet.Utils;
using Numpy;

namespace MLNet.Regression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public abstract class AbstractLinearRegression : LinearModel
    {
        public enum SloveFuc
        {
            Slove,
            SGD
        }

        protected AbstractLinearRegression(string name)
            : base(name)
        {
        }

        internal abstract Func<NDarray, NDarray, NDarray, NDarray> LeastSquares { set; get; }

        public NDarray? TheDa { set; get; }

        public SloveFuc SloveFunc { set; get; }


        internal override void fit(NDarray x, NDarray y, double learning_rate)
        {
            switch (SloveFunc)
            {
                case SloveFuc.Slove:
                    TheDa = Slove(x, y);
                    break;
                case SloveFuc.SGD:
                    TheDa = sgd(x, y, learning_rate);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            Log.Print?.Invoke($"{Name} Fit:\r\n{TheDa}");
        }

        public abstract NDarray Slove(NDarray x, NDarray y);


        internal override NDarray sgd(NDarray x, NDarray y, double learning_rate)
        {
            if (LeastSquares == null)
                throw new Exception("Not define Loss function");

            // x => 1,x1,x2,x3,...,xN
            var X = np2.linear_first_order(x);
            var theDa = np.random.randn(X.shape[1], 1);
            Enumerable.Range(0, 100).ToList().ForEach(epoch =>
            {
                Log.Print?.Invoke($"{Name} Epoch:\r\n{epoch}");
                var pred = predict(X, theDa);

                var cost = np2.power(pred - y, 2);
            });
            return null;
        }

        public double Cost(NDarray x, NDarray y, NDarray theda)
        {
            var cost = predict(x, theda);
            var power = np.ones_like(cost) * 2;
            return np.average(np.power(cost - y, power));
        }


        internal NDarray predict(NDarray x, NDarray theda)
        {
            return np.matmul(x, theda);
        }

        internal override NDarray call(NDarray x)
        {
            return np.matmul(x, TheDa);
        }


        public override void Load(string path)
        {
        }

        public override void Save(string path)
        {
        }
    }
}