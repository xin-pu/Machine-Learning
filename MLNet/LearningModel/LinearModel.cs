using Numpy;

namespace MLNet.LearningModel
{
    public abstract class LinearModel : Model
    {
        protected LinearModel(string name)
            : base(name)
        {
        }


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


        public abstract void Fit(NDarray x, NDarray y);

        public abstract NDarray SGD(NDarray x, NDarray y);

        public abstract NDarray Pred(NDarray x);
    }
}