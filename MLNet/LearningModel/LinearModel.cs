using MLNet.Utils;
using Numpy;

namespace MLNet.LearningModel
{
    public abstract class LinearModel : Model
    {
        protected LinearModel(string name,
            PrimaryType primaryType = PrimaryType.Polynomial,
            int alpha = 16)
            : base(name)
        {
            PrimaryType = primaryType;
            Alpha = alpha;
        }


        public PrimaryType PrimaryType { set; get; }

        public int Alpha { set; get; }

        public NDarray? Theda { set; get; }

        public void Fit(double[,] x, double[] y)
        {
            var X = ConvertX(x);
            var Y = np.expand_dims(np.array(y), -1);
            Theda = fit(X, Y);
        }


        public override void Save(string path)
        {
        }

        public override void Load(string path)
        {
        }


        public virtual double Evaluate(double[,] x, double[] y)
        {
            return double.NaN;
        }

        public virtual double Predict(double[] X)
        {
            return double.NaN;
        }


        internal abstract NDarray fit(NDarray X, NDarray Y);

        internal NDarray ConvertX(double[,] x)
        {
            var batch = x.GetLength(0);
            var dims = x.GetLength(1);
            var res = np.zeros(batch, Alpha);
            Enumerable.Range(0, batch).ToList().ForEach(i =>
            {
                res[i] = np.array(PrimaryFunc.getPolyPrimary(x[i, 0], Alpha));
            });
            return res;
        }
    }
}