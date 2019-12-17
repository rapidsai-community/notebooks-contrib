{%- extends 'python.tpl' -%}

{%- block  header -%}
#necessary for Dask support
from multiprocessing import freeze_support
import os

timeout = 3600.0

def time_notebook():
{%endblock header%}

{%- block codecell -%}
  {%- if cell.source -%}
     {%-if cell.source.startswith('%matplotlib') -%}
        {#Skipping matplotlib cell #}
     {%- elif cell.source.find('%tim')>=0 -%}
        {{cell.source|replace('%%','%')|replace('%timeit ',"")|replace('%time ',"")|ipython2python|indent(4)}}
     {%- elif cell.source.startswith("!") -%}
	{{cell.source|ipython2python|indent(4)|replace('get_ipython().system','os.system')}}
     {%- else -%}
        {{cell.source|ipython2python|indent(4)}}
     {%- endif -%}
  {%- endif -%}
{% endblock codecell %}

{%- block markdowncell %}
{% endblock markdowncell -%}

{%block footer%}
if __name__ == '__main__':
   freeze_support()
   time_notebook()
{%endblock footer%}