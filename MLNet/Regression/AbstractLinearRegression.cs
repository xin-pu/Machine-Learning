using MLNet.LearningModel;
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

        public NDarray? TheDa { set; get; }

        public SloveFuc SloveFunc { set; get; }


        public override void Fit(NDarray x, NDarray y)
        {
            switch (SloveFunc)
            {
                case SloveFuc.Slove:
                    TheDa = Slove(x, y);
                    break;
                case SloveFuc.SGD:
                    TheDa = SGD(x, y);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            Log.Print?.Invoke($"{Name} Fit:\r\n{TheDa}");
        }

        public abstract NDarray Slove(NDarray x, NDarray y);

        public override NDarray SGD(NDarray x, NDarray y)
        {
            return null;
        }

        public override NDarray Pred(NDarray x)
        {
            return x.multiply(TheDa);
        }
    }
}