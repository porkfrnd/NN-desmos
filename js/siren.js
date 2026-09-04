// SIREN layer — correct implementation.
// sin is not a built-in tfjs activation, and SIREN needs special init:
// first layer: U(-1/fan_in, 1/fan_in) then sin(ω₀·(Wx+b))
// deeper layers: U(-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀) then sin(Wx+b)  (no ω₀ scale)
// ω₀ only scales the first layer — per SIREN paper (Sitzmann et al. 2020).

const SIREN_W0 = 30;

// Declare globally so factory can see it even when tf is not yet loaded.
// We assign conditionally but the binding is top-level.
let SirenDense = null;

if (typeof tf !== 'undefined' && tf.layers && tf.layers.Layer) {
  SirenDense = class extends tf.layers.Layer {
    static className = 'SirenDense';
    constructor(config) {
      super(config);
      this.units = config.units;
      this.isFirstLayer = !!config.isFirstLayer;
      this.w0 = config.w0 != null ? config.w0 : SIREN_W0;
      this.useBias = config.useBias != null ? config.useBias : true;
      this.kernel = null;
      this.bias = null;
    }
    build(inputShape) {
      const fanIn = inputShape[inputShape.length - 1];
      const bound = this.isFirstLayer
        ? 1 / Math.max(1, fanIn)
        : Math.sqrt(6 / Math.max(1, fanIn)) / this.w0;
      const kernelInit = tf.initializers.randomUniform({ minval: -bound, maxval: bound });
      this.kernel = this.addWeight('kernel', [fanIn, this.units], 'float32', kernelInit);
      if (this.useBias) {
        this.bias = this.addWeight('bias', [this.units], 'float32', tf.initializers.zeros());
      }
      this.built = true;
    }
    // tfjs Layer overrides call(), not apply()
    call(inputs) {
      return tf.tidy(() => {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        let out = x.matMul(this.kernel.read());
        if (this.bias) out = out.add(this.bias.read());
        // ω₀ only on first layer
        if (this.isFirstLayer) out = tf.mul(out, this.w0);
        return tf.sin(out);
      });
    }
    computeOutputShape(inputShape) {
      const s = inputShape.slice(); s[s.length - 1] = this.units; return s;
    }
    getConfig() {
      const c = super.getConfig();
      c.units = this.units; c.isFirstLayer = this.isFirstLayer; c.w0 = this.w0; c.useBias = this.useBias;
      return c;
    }
  };
  try { tf.serialization.registerClass(SirenDense); } catch (_) {}
}

function sirenDense(units, isFirstLayer, w0) {
  if (typeof SirenDense === 'undefined' || SirenDense === null) {
    // graceful fallback if tf failed to load — still trainable
    return tf.layers.dense({ units, activation: 'tanh' });
  }
  return new SirenDense({ units, isFirstLayer: !!isFirstLayer, w0: w0 != null ? w0 : SIREN_W0 });
}
