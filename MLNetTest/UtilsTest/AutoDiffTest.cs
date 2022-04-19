using System;
using AutoDiff;
using FluentAssertions;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.UtilsTest
{
    public class AutoDiffTest : AbstractUnitTest
    {
        private readonly Func<double, double, double> Func = (x, y) =>
            (x + y) * Math.Log(Math.Pow(Math.E, x) + Math.Pow(Math.E, y));

        public AutoDiffTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        /// <summary>
        ///     f(x,y)=(x+y)Log(e^x+e^y)
        /// </summary>
        [Fact]
        public void diff()
        {
            var x = new Variable();
            var y = new Variable();
            var func = (x + y) * TermBuilder.Log(TermBuilder.Exp(x) + TermBuilder.Exp(y));


            var point = new double[] { 2, -1 };
            var variables = new[] { x, y };
            var value = func.Evaluate(variables, point);
            var grad = func.Differentiate(variables, point);

            print(value);
            print(grad);

            var value_true = Func(2, -1);
            print(value_true);
            value_true.Should().Be(value);
        }
    }
}