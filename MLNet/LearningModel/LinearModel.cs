using MLNet.Utils;
using Numpy;

namespace MLNet.LearningModel
{
    public abstract class LinearModel : Model
    {
        protected LinearModel(string name,
            PrimaryType primaryType = PrimaryType.Polynomial,
            int alpha = 1)
            : base(name)
        {
            PrimaryType = primaryType;
            Alpha = alpha;
        }


        public PrimaryType PrimaryType { set; get; }

        public MultiPrimaryType MultiPrimaryType { set; get; }

        public int Alpha { set; get; }

        public NDarray? Theda { set; get; }


        public override void Save(string path)
        {
        }

        public override void Load(string path)
        {
        }

        public void Slove(double[,] X, double[,] Y)
        {
            var npX = PrimaryExpand(X, PrimaryType, Alpha);
            var npY = PrimaryExpand(Y, PrimaryType.Original, 1);
            Theda = slove(npX, npY);
        }

        protected void Slove(NDarray X, NDarray Y)
        {
            Theda = slove(X, Y);
        }

        public void Fit(double[,] X, double[,] Y)
        {
            var npX = PrimaryExpand(X, PrimaryType, Alpha);
            var npY = PrimaryExpand(Y, PrimaryType.Original, 1);
            Theda = slove(npX, npY);
        }

        protected void Fit(NDarray X, NDarray Y)
        {
            Theda = fit(X, Y);
        }

        public NDarray Predict(double[,] x)
        {
            var npX = PrimaryExpand(x, PrimaryType, Alpha);
            return np.matmul(npX, Theda);
        }


        internal abstract NDarray slove(NDarray X, NDarray Y);

        internal abstract NDarray fit(NDarray X, NDarray Y);

        private NDarray PrimaryExpand(double[,] data, PrimaryType primaryType, int alpha)
        {
            var dims = data.GetLength(1);
            if (dims == 1) return SinglePrimaryExpand(data, primaryType, alpha);
            return MultiPrimaryExpand(data, primaryType, alpha);
        }

        private NDarray SinglePrimaryExpand(double[,] data, PrimaryType primaryType, int alpha)
        {
            var batch = data.GetLength(0);
            var res = np.zeros(batch, alpha);
            Enumerable.Range(0, batch).AsParallel().ToList().ForEach(i =>
            {
                switch (primaryType)
                {
                    case PrimaryType.Original:
                        res[i] = np.array(data[i, 0]);
                        break;
                    case PrimaryType.Polynomial:
                        res[i] = np.array(PrimaryFunc.getPolyPrimary(data[i, 0], alpha));
                        break;
                    case PrimaryType.Triangle:
                        res[i] = np.array(PrimaryFunc.getTrigPrimary(data[i, 0], alpha));
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(primaryType), primaryType, null);
                }
            });
            return res;
        }

        private NDarray MultiPrimaryExpand(double[,] data, PrimaryType primaryType, int alpha)
        {
            throw new NotImplementedException();
        }
    }
}