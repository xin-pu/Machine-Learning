using Numpy;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class PolynomialFeaturesTest : AbstractUnitTest
    {
        public PolynomialFeaturesTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            var x = np.arange(-3, 3, 0.1);
            X = np.expand_dims(x, -1);
            Y = np.expand_dims(1 + 0.3 * np.power(x, np.array(3)), -1);
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        //[Fact]
        //public void PolynomialFeatures()
        //{
        //    var pf = new PolynomialFeatures(4)
        //    {
        //        Print = false
        //    };
        //    pf.Fit(X, Y, 1E-3, 5000);

        //    print(pf.Resolve);


        //    print(Y);
        //    print(pf.Call(X));
        //}


        //[Fact]
        //public void LassoPolynomialFeatures()
        //{
        //    var ridge = new PolynomialFeatures(4)
        //    {
        //        Constraint = Constraint.L1,
        //        Print = false
        //    };

        //    ridge.Fit(X, Y, 1E-3, 5000);
        //    ridge.PrintSelf();
        //}

        //[Fact]
        //public void RidgePolynomialFeatures()
        //{
        //    var ridge = new PolynomialFeatures(4)
        //    {
        //        Constraint = Constraint.L2,
        //        Print = false
        //    };

        //    ridge.Fit(X, Y, 1E-3, 5000);
        //    ridge.PrintSelf();
        //}

        //[Fact]
        //public void ElasticNetPolynomialFeatures()
        //{
        //    var ridge = new PolynomialFeatures(4)
        //    {
        //        Constraint = Constraint.ElasticNet,
        //        Print = false
        //    };

        //    ridge.Fit(X, Y, 1E-3, 5000);
        //    ridge.PrintSelf();
        //}


        //[Fact]
        //public void TrianglePolynomialFeatures()
        //{
        //    var pf = new TrianglePolynomialFeatures(15)
        //    {
        //        Constraint = Constraint.ElasticNet,
        //        Print = false
        //    };
        //    pf.Fit(X, Y, 1E-3, 5000);
        //    print(pf.Resolve);
        //    ;

        //    print(Y);
        //    print(pf.Call(X));
        //}
    }
}