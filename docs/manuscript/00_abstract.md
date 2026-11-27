# Abstract

PyMCP exposes selected PyMDP operations through a shared registry reached from
MCP, REST and a direct in-process client [@repository; @heins2022]. We describe the
conditions that make a discrete-state run interpretable across those entry points.
The contract admits a finite supported model, makes action selection and transition
commit explicit, and distinguishes requested work from transitions actually
executed. Published checks exercise numerical oracles and real transports.
This report documents observed repository contracts and their coverage boundaries.
