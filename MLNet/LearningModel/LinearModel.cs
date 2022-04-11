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


        public override void Save(string path)
        {
        }

        public override void Load(string path)
        {
        }


        internal abstract NDarray fit(NDarray X, NDarray Y);
    }
}