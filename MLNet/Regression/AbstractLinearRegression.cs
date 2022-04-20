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

        internal virtual Term? CreateLoss(Variable[] w, NDarray x, NDarray y)
        {
            if (TheDa == null) return null;
            var batchsize = x.shape[0];
            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i},:"].GetData<double>();
                var rowY = y[$"{i},:"].GetData<double>();
                return TermBuilder.Power(rowX[0] * w[0] + rowX[1] * w[1] - rowY[0], 2);
            });
            return TermBuilder.Sum(list) / batchsize;
        }

        internal abstract NDarray slove(NDarray x, NDarray y);

        internal override NDarray sgd(NDarray x, NDarray y, double learning_rate)
        {
            var theDa = np.random.randn(x.shape[1], 1);

            var w = new[] {new Variable(), new Variable()};

            Loss = CreateLoss(w, x, y);

            Enumerable.Range(0, 100).ToList().ForEach(epoch =>
            {
                Log.print?.Invoke($"{Name} Epoch:\t{epoch}\tStart");

                var theda = theDa?.GetData<double>();

                var loss = Loss.Evaluate(w, theda);
                var gradarray = Loss.Differentiate(w, theda);
                var grad = np.expand_dims(np.array(gradarray), -1);
                theDa -= learning_rate * grad;

                Log.print?.Invoke($"{Name} Epoch:\t{epoch:D2}\tLoss:{loss:F4}\r\n");
            });
            return theDa;
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