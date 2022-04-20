using AutoDiff;
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

        internal Term? Loss { set; get; }

        public NDarray? TheDa { set; get; }

        public SloveFuc SloveFunc { set; get; } = SloveFuc.SGD;


        internal override void fit(NDarray x, NDarray y, double learning_rate)
        {
            // x => 1,x1,x2,x3,...,xN
            var X = np2.linear_first_order(x);
            TheDa = np.random.randn(X.shape[1], 1);
            Loss = CreateLoss();

            switch (SloveFunc)
            {
                case SloveFuc.Slove:
                    TheDa = slove(X, y);
                    break;
                case SloveFuc.SGD:
                    TheDa = sgd(X, y, learning_rate);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        internal virtual Term? CreateLoss()
        {
            if (TheDa == null) return null;
            var height = TheDa.shape[0];

            var b = TheDa["0,:"].GetData<double>()[0];
            var term = (Term) b;

            foreach (var i in Enumerable.Range(1, height))
            {
                var x_i = new Variable();
                var w_i = TheDa[$"{i},:"].GetData<double>()[0];
                term += w_i * x_i;
            }

            return term;
        }

        internal abstract NDarray slove(NDarray x, NDarray y);

        internal override NDarray sgd(NDarray x, NDarray y, double learning_rate)
        {
            var theDa = np.random.randn(x.shape[1], 1);
            Enumerable.Range(0, 100).ToList().ForEach(epoch =>
            {
                Log.print?.Invoke($"{Name} Epoch:\t{epoch}\tStart");
                var pred = predict(x, theDa);
                var cost = np2.power(pred - y, 2);


                Log.print?.Invoke($"{Name} Epoch:\t{epoch}\tLoss{cost}\r\n");
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