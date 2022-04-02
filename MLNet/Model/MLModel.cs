using System.Text;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MLNet.Model;

public abstract class MLModel
{
    protected MLModel(string name)
    {
        Name = name;
    }

    public string Name { get; set; }

    public abstract void Fit(Matrix X, Matrix Y);

    public abstract double Evaluate(Matrix X, Matrix Y);

    public abstract Matrix Predict(Matrix X);

    public abstract void Save(string path);

    public abstract MLModel Load(string path);

    public override string ToString()
    {
        var strBuild = new StringBuilder();
        strBuild.AppendLine($"Name");
        return strBuild.ToString();
    }
}