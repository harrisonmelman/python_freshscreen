"""Neuroglancer shader source for image layers."""


def grayscale_shader(color: str = "white"):
    """single-channel shader: window/level, gamma, invert, a tint color, and an optional transparent background for overlays.
    Expects shaderControls {"normalized": {"range": [...], "window": [...]}}"""
    return (
        "#uicontrol invlerp normalized(range=[0, 0.8], clamp=true)\n"
        "#uicontrol float log2_gamma slider(min=-2, max=2, default=0, step=0.05)\n"
        "#uicontrol bool invert checkbox(default=false)\n"
        f"#uicontrol vec3 color color(default=\"{color}\")\n"
        "#uicontrol bool transparent_bg checkbox(default=false)\n\n"
        "void main() {\n"
        "  float x = clamp(normalized(), 0.0, 1.0);\n"
        "  if (invert) x = 1.0 - x;\n"
        "  x = pow(x, exp2(-log2_gamma));   // log2_gamma > 0 brightens, < 0 darkens\n"
        "  if (transparent_bg) {\n"
        "    emitRGBA(vec4(color * x, x));\n"
        "  } else {\n"
        "    emitRGB(color * x);\n"
        "  }\n"
        "}"
    )


# 3-channel RGB shader for color volumes (gqi-color, tdi-color, ...)
# Expects shaderControls {"black": ..., "white": ...}
RGB_SHADER = """
    #uicontrol float black slider(min=0, max=1, default=0)
    #uicontrol float white slider(min=0, max=1, default=1)
    #uicontrol float gamma slider(min=0.2, max=3, default=1)
    #uicontrol float saturation slider(min=0, max=3, default=1)
    #uicontrol float opacity slider(min=0, max=1, default=1)
    void main() {
    vec3 rgb = vec3(toNormalized(getDataValue(0)),
                    toNormalized(getDataValue(1)),
                    toNormalized(getDataValue(2)));
    rgb = clamp((rgb - black) / max(white - black, 1e-6), 0.0, 1.0);
    rgb = pow(rgb, vec3(1.0 / gamma));
    float lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    rgb = clamp(mix(vec3(lum), rgb, saturation), 0.0, 1.0);
    emitRGBA(vec4(rgb, opacity));
    }"""
