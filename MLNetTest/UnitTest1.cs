using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using MLNet.Regression.LinearRegression;
using MLNet.Utils;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest;

public class UnitTest1 : AbstractUnitTest
{
    public UnitTest1(ITestOutputHelper testOutputHelper)
        : base(testOutputHelper)
    {
    }


    private Func<double, double> func =>
        x => 2 * Math.Sin(2 * x) + Math.Pow(x, 3) + SystemRandomSource.NextDouble() * 0.02;


    [Fact]
    public void LRTest()
    {
        var model = new NormalEquationsLR();
        var x = EnumerableExt.GetList(-3, 3, 50);
        var y = x.Select(func).ToArray();

        var X = (Matrix)PrimaryFunc.getPolyPrimaryS(x.ToArray(), 8);
        var Y = (Matrix)Matrix.Build.DenseOfColumnArrays(y);
        model.Fit(X, Y);
    }
}