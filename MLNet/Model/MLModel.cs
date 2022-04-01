using MathNet.Numerics.LinearAlgebra.Double;
using System.Text;

namespace MLNet.Model
{
    public abstract class MLModel
    {
        public MLModel(string name)
        {
            Name= name;
        }

        public string Name { get; set; }

        public abstract void Fit(Matrix X, Matrix Y);

        public abstract double Evaluate(Matrix X, Matrix Y);

        public abstract double Predict(Matrix X, Matrix Y);

        public abstract void Save(string path);

        public abstract MLModel Load(string path);

        public override string ToString()
        {
            var strBuild = new StringBuilder();
            strBuild.AppendLine($"Name");
            return strBuild.ToString();
        }
    }
}
