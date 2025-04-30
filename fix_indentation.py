print("Fixing app.py indentation")
with open('app.py', 'r') as f:
    lines = f.readlines()

# Fix the indentation of the except block at line 1171
if '    except Exception as e:' in lines[1170]:
    lines[1170] = lines[1170].replace('    except Exception as e:', '        except Exception as e:')

# Fix other indentation issues in the Root Cause Analysis section
if 'return' in lines[1183]:
    lines[1183] = lines[1183].replace('return', '                return')

# Add missing except block for the try at line 897
try_block_line = 897
has_matching_except = False
for i in range(try_block_line, len(lines)):
    if 'except Exception as e:' in lines[i] and lines[i].startswith('    except'):
        has_matching_except = True
        break
    if 'def ' in lines[i] and lines[i].startswith('def '):
        break

if not has_matching_except:
    # Find a good place to add the except block
    for i in range(1540, 1550):
        if 'def display_visualizations' in lines[i]:
            # Add the except block before the function definition
            lines[i] = '    except Exception as e:
        logger.error(f"Error in detailed analysis: {str(e)}", exc_info=True)
        st.error(f"Error in detailed analysis: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)

' + lines[i]
            break

with open('app_fixed_new.py', 'w') as f:
    f.writelines(lines)

print('Fixed file written to app_fixed_new.py')
