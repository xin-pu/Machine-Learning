using MLNet.Transforms;

namespace MLNet.Models.Regression
{
    public class TrianglePolyRegression : PolyRegression
    {
        public TrianglePolyRegression(
            int degree = 1)
        {
            Degree = degree;
            Transform = new TrianglePoly(Degree);
        }
    }
}