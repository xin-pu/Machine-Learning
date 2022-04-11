using Numpy;

namespace MLNet.LearningModel
{
    public abstract class LinearModel : Model
    {
        protected LinearModel(string name)
            : base(name)
        {
        }

        public NDarray? Theda { set; get; }

        public SloveFuc SloveFunc { set; get; }

        public override void Save(string path)
        {
        }

        public override void Load(string path)
        {
        }


        public virtual NDarray Predict(NDarray x)
        {
            var res = Pred(x);
            Log.Print?.Invoke($"{Name} Predict:\r\n{res}");
            return res;
        }


        public virtual void Fit(NDarray x, NDarray y)
        {
            switch (SloveFunc)
            {
                case SloveFuc.Slove:
                    Theda = Slove(x, y);
                    break;
                case SloveFuc.SGD:
                    Theda = SGD(x, y);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            Log.Print?.Invoke($"{Name} Fit:\r\n{Theda}");
        }

        public abstract NDarray Slove(NDarray x, NDarray y);

        public abstract NDarray SGD(NDarray x, NDarray y);

        public abstract NDarray Pred(NDarray x);
    }

    public enum SloveFuc
    {
        Slove,
        SGD
    }
}