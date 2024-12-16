import os

# Directory containing the textures
texture_directory = "/home/emlyn/texture_pngs"

# Output file to save the XML
output_file = "texture_pngs_output.xml"

# XML format for each texture
texture_template = '<texture name="{name}" file="textures/{filename}" type="2d"/>'

# Gather all texture files
textures = [f for f in os.listdir(texture_directory) if f.endswith('.png')]

# Generate XML for each texture
texture_xml = []
for texture in textures:
    # Extract the base name without extension and remove extra identifiers
    base_name = texture.split("_")[0]  # Keep only the meaningful part of the name
    texture_entry = texture_template.format(name=base_name, filename=texture)
    texture_xml.append(texture_entry)

# Save the XML to a file
with open(output_file, "w") as f:
    f.write("\n".join(texture_xml))

print(f"XML entries written to {output_file}")
